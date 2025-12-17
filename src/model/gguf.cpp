#include "granite/gguf.h"
#include "granite/backend.h"
#include "granite/tensor.h"
#include "granite/log.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <algorithm>

namespace granite {

// =============================================================================
// GGML Type Utilities
// =============================================================================

const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGMLType::F32:     return "F32";
        case GGMLType::F16:     return "F16";
        case GGMLType::Q4_0:    return "Q4_0";
        case GGMLType::Q4_1:    return "Q4_1";
        case GGMLType::Q5_0:    return "Q5_0";
        case GGMLType::Q5_1:    return "Q5_1";
        case GGMLType::Q8_0:    return "Q8_0";
        case GGMLType::Q8_1:    return "Q8_1";
        case GGMLType::Q2_K:    return "Q2_K";
        case GGMLType::Q3_K:    return "Q3_K";
        case GGMLType::Q4_K:    return "Q4_K";
        case GGMLType::Q5_K:    return "Q5_K";
        case GGMLType::Q6_K:    return "Q6_K";
        case GGMLType::Q8_K:    return "Q8_K";
        case GGMLType::IQ4_NL:  return "IQ4_NL";
        case GGMLType::IQ3_S:   return "IQ3_S";
        case GGMLType::IQ4_XS:  return "IQ4_XS";
        case GGMLType::BF16:    return "BF16";
        case GGMLType::I8:      return "I8";
        case GGMLType::I16:     return "I16";
        case GGMLType::I32:     return "I32";
        case GGMLType::I64:     return "I64";
        case GGMLType::F64:     return "F64";
        default:                return "UNKNOWN";
    }
}

size_t ggml_type_size(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return 4;
        case GGMLType::F16:  return 2;
        case GGMLType::BF16: return 2;
        case GGMLType::I8:   return 1;
        case GGMLType::I16:  return 2;
        case GGMLType::I32:  return 4;
        case GGMLType::I64:  return 8;
        case GGMLType::F64:  return 8;
        default:             return 0;  // Quantized types use block size
    }
}

size_t ggml_type_block_size(GGMLType type) {
    switch (type) {
        case GGMLType::F32:
        case GGMLType::F16:
        case GGMLType::BF16:
        case GGMLType::I8:
        case GGMLType::I16:
        case GGMLType::I32:
        case GGMLType::I64:
        case GGMLType::F64:
            return 1;

        case GGMLType::Q4_0:
        case GGMLType::Q4_1:
        case GGMLType::Q5_0:
        case GGMLType::Q5_1:
        case GGMLType::Q8_0:
        case GGMLType::Q8_1:
        case GGMLType::IQ4_NL:
            return 32;

        case GGMLType::Q2_K:
        case GGMLType::Q3_K:
        case GGMLType::Q4_K:
        case GGMLType::Q5_K:
        case GGMLType::Q6_K:
        case GGMLType::Q8_K:
        case GGMLType::IQ3_S:
        case GGMLType::IQ4_XS:
            return 256;

        default:
            return 32;
    }
}

size_t ggml_type_bytes_per_block(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return 4;
        case GGMLType::F16:  return 2;
        case GGMLType::BF16: return 2;
        case GGMLType::I8:   return 1;
        case GGMLType::I16:  return 2;
        case GGMLType::I32:  return 4;
        case GGMLType::I64:  return 8;
        case GGMLType::F64:  return 8;

        // Q4_0: 32 elements, 2 bytes scale + 16 bytes data = 18 bytes
        case GGMLType::Q4_0: return 18;

        // Q4_1: 32 elements, 2 bytes scale + 2 bytes min + 16 bytes data = 20 bytes
        case GGMLType::Q4_1: return 20;

        // Q5_0: 32 elements, 2 bytes scale + 4 bytes high bits + 16 bytes low = 22 bytes
        case GGMLType::Q5_0: return 22;

        // Q5_1: 32 elements, 2 bytes scale + 2 bytes min + 4 bytes high + 16 bytes = 24 bytes
        case GGMLType::Q5_1: return 24;

        // Q8_0: 32 elements, 2 bytes scale + 32 bytes data = 34 bytes
        case GGMLType::Q8_0: return 34;

        // Q8_1: 32 elements, 4 bytes scale + 4 bytes sum + 32 bytes data = 40 bytes
        case GGMLType::Q8_1: return 40;

        // K-quants (256 element blocks)
        case GGMLType::Q2_K: return 84;
        case GGMLType::Q3_K: return 110;
        case GGMLType::Q4_K: return 144;
        case GGMLType::Q5_K: return 176;
        case GGMLType::Q6_K: return 210;
        case GGMLType::Q8_K: return 292;

        // IQ4_NL: 32 elements, 2 bytes scale + 16 bytes data = 18 bytes
        case GGMLType::IQ4_NL: return 18;

        // IQ4_XS: 256 elements, 2 bytes d + 2 bytes scales_h + 4 bytes scales_l + 128 bytes qs = 136 bytes
        case GGMLType::IQ4_XS: return 136;

        // IQ3_S: 256 elements, 2 bytes d + 64 bytes qs + 8 bytes qh + 32 bytes signs + 4 bytes scales = 110 bytes
        case GGMLType::IQ3_S: return 110;

        default:
            return 0;
    }
}

bool ggml_type_is_quantized(GGMLType type) {
    switch (type) {
        case GGMLType::F32:
        case GGMLType::F16:
        case GGMLType::BF16:
        case GGMLType::I8:
        case GGMLType::I16:
        case GGMLType::I32:
        case GGMLType::I64:
        case GGMLType::F64:
            return false;
        default:
            return true;
    }
}

std::optional<DataType> ggml_type_to_dtype(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return DataType::FP32;
        case GGMLType::F16:  return DataType::FP16;
        case GGMLType::BF16: return DataType::BF16;
        case GGMLType::I8:   return DataType::INT8;
        case GGMLType::I16:  return DataType::INT16;
        case GGMLType::I32:  return DataType::INT32;
        default:             return std::nullopt;  // Quantized types need dequantization
    }
}

// =============================================================================
// GGUFTensorInfo
// =============================================================================

size_t GGUFTensorInfo::numel() const {
    if (dimensions.empty()) return 1;
    size_t n = 1;
    for (auto d : dimensions) n *= d;
    return n;
}

size_t GGUFTensorInfo::size_bytes() const {
    size_t n = numel();
    size_t block_size = ggml_type_block_size(type);
    size_t bytes_per_block = ggml_type_bytes_per_block(type);

    if (block_size == 1) {
        return n * bytes_per_block;
    } else {
        // Quantized: round up to block boundary
        size_t num_blocks = (n + block_size - 1) / block_size;
        return num_blocks * bytes_per_block;
    }
}

// =============================================================================
// GGUFFile Implementation
// =============================================================================

GGUFFile::~GGUFFile() {
    if (data_ != nullptr) {
        munmap(data_, file_size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

GGUFFile::GGUFFile(GGUFFile&& other) noexcept
    : path_(std::move(other.path_))
    , header_(other.header_)
    , metadata_(std::move(other.metadata_))
    , tensors_(std::move(other.tensors_))
    , data_(other.data_)
    , file_size_(other.file_size_)
    , data_offset_(other.data_offset_)
    , fd_(other.fd_)
{
    other.data_ = nullptr;
    other.fd_ = -1;
}

GGUFFile& GGUFFile::operator=(GGUFFile&& other) noexcept {
    if (this != &other) {
        if (data_ != nullptr) {
            munmap(data_, file_size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }

        path_ = std::move(other.path_);
        header_ = other.header_;
        metadata_ = std::move(other.metadata_);
        tensors_ = std::move(other.tensors_);
        data_ = other.data_;
        file_size_ = other.file_size_;
        data_offset_ = other.data_offset_;
        fd_ = other.fd_;

        other.data_ = nullptr;
        other.fd_ = -1;
    }
    return *this;
}

Result<GGUFFile> GGUFFile::open(const std::string& path) {
    GGUFFile file;
    file.path_ = path;

    // Open file
    file.fd_ = ::open(path.c_str(), O_RDONLY);
    if (file.fd_ < 0) {
        GRANITE_FAIL(ErrorCode::FileNotFound,
                     fmt::format("Failed to open file: {}", path));
    }

    // Get file size
    struct stat st;
    if (fstat(file.fd_, &st) < 0) {
        close(file.fd_);
        GRANITE_FAIL(ErrorCode::IOError, "Failed to get file size");
    }
    file.file_size_ = static_cast<size_t>(st.st_size);

    // Memory map the file
    file.data_ = mmap(nullptr, file.file_size_, PROT_READ, MAP_PRIVATE, file.fd_, 0);
    if (file.data_ == MAP_FAILED) {
        close(file.fd_);
        file.data_ = nullptr;
        GRANITE_FAIL(ErrorCode::IOError, "Failed to memory map file");
    }

    // Parse file contents
    auto header_result = file.parse_header();
    if (!header_result.ok()) {
        return header_result.error();
    }

    auto metadata_result = file.parse_metadata();
    if (!metadata_result.ok()) {
        return metadata_result.error();
    }

    auto tensors_result = file.parse_tensors();
    if (!tensors_result.ok()) {
        return tensors_result.error();
    }

    GRANITE_LOG_INFO("Opened GGUF file: {} (v{}, {} tensors, {} metadata)",
                     path, file.header_.version,
                     file.header_.tensor_count,
                     file.header_.metadata_kv_count);

    return file;
}

Result<void> GGUFFile::parse_header() {
    if (file_size_ < sizeof(GGUFHeader)) {
        GRANITE_FAIL(ErrorCode::InvalidFormat, "File too small for GGUF header");
    }

    auto* ptr = static_cast<const uint8_t*>(data_);

    // Read magic
    memcpy(&header_.magic, ptr, 4);
    ptr += 4;

    if (header_.magic != GGUF_MAGIC) {
        GRANITE_FAIL(ErrorCode::InvalidFormat,
                     fmt::format("Invalid GGUF magic: 0x{:08X}", header_.magic));
    }

    // Read version
    memcpy(&header_.version, ptr, 4);
    ptr += 4;

    if (header_.version < GGUF_VERSION_MIN || header_.version > GGUF_VERSION_MAX) {
        GRANITE_FAIL(ErrorCode::InvalidFormat,
                     fmt::format("Unsupported GGUF version: {}", header_.version));
    }

    // Read counts
    memcpy(&header_.tensor_count, ptr, 8);
    ptr += 8;

    memcpy(&header_.metadata_kv_count, ptr, 8);

    return {};
}

std::string GGUFFile::read_string(size_t& offset) {
    auto* ptr = static_cast<const uint8_t*>(data_) + offset;

    uint64_t len;
    memcpy(&len, ptr, 8);
    offset += 8;

    std::string result(reinterpret_cast<const char*>(ptr + 8 - offset + offset), len);
    offset += len;

    return result;
}

Result<GGUFValue> GGUFFile::read_value(GGUFType type, size_t& offset) {
    auto* ptr = static_cast<const uint8_t*>(data_) + offset;

    switch (type) {
        case GGUFType::UINT8: {
            uint8_t val;
            memcpy(&val, ptr, 1);
            offset += 1;
            return GGUFValue(val);
        }
        case GGUFType::INT8: {
            int8_t val;
            memcpy(&val, ptr, 1);
            offset += 1;
            return GGUFValue(val);
        }
        case GGUFType::UINT16: {
            uint16_t val;
            memcpy(&val, ptr, 2);
            offset += 2;
            return GGUFValue(val);
        }
        case GGUFType::INT16: {
            int16_t val;
            memcpy(&val, ptr, 2);
            offset += 2;
            return GGUFValue(val);
        }
        case GGUFType::UINT32: {
            uint32_t val;
            memcpy(&val, ptr, 4);
            offset += 4;
            return GGUFValue(val);
        }
        case GGUFType::INT32: {
            int32_t val;
            memcpy(&val, ptr, 4);
            offset += 4;
            return GGUFValue(val);
        }
        case GGUFType::FLOAT32: {
            float val;
            memcpy(&val, ptr, 4);
            offset += 4;
            return GGUFValue(val);
        }
        case GGUFType::BOOL: {
            bool val = (*ptr != 0);
            offset += 1;
            return GGUFValue(val);
        }
        case GGUFType::STRING: {
            return GGUFValue(read_string(offset));
        }
        case GGUFType::UINT64: {
            uint64_t val;
            memcpy(&val, ptr, 8);
            offset += 8;
            return GGUFValue(val);
        }
        case GGUFType::INT64: {
            int64_t val;
            memcpy(&val, ptr, 8);
            offset += 8;
            return GGUFValue(val);
        }
        case GGUFType::FLOAT64: {
            double val;
            memcpy(&val, ptr, 8);
            offset += 8;
            return GGUFValue(val);
        }
        case GGUFType::ARRAY: {
            uint32_t elem_type;
            memcpy(&elem_type, ptr, 4);
            offset += 4;

            uint64_t count;
            memcpy(&count, static_cast<const uint8_t*>(data_) + offset, 8);
            offset += 8;

            // Handle common array types
            if (static_cast<GGUFType>(elem_type) == GGUFType::INT32) {
                std::vector<int32_t> arr(count);
                for (uint64_t i = 0; i < count; i++) {
                    memcpy(&arr[i], static_cast<const uint8_t*>(data_) + offset, 4);
                    offset += 4;
                }
                return GGUFValue(arr);
            } else if (static_cast<GGUFType>(elem_type) == GGUFType::FLOAT32) {
                std::vector<float> arr(count);
                for (uint64_t i = 0; i < count; i++) {
                    memcpy(&arr[i], static_cast<const uint8_t*>(data_) + offset, 4);
                    offset += 4;
                }
                return GGUFValue(arr);
            } else if (static_cast<GGUFType>(elem_type) == GGUFType::STRING) {
                std::vector<std::string> arr;
                arr.reserve(count);
                for (uint64_t i = 0; i < count; i++) {
                    arr.push_back(read_string(offset));
                }
                return GGUFValue(arr);
            } else {
                // Generic uint8 array for unsupported types
                std::vector<uint8_t> arr(count);
                for (uint64_t i = 0; i < count; i++) {
                    arr[i] = *(static_cast<const uint8_t*>(data_) + offset);
                    offset += 1;
                }
                return GGUFValue(arr);
            }
        }
        default:
            GRANITE_FAIL(ErrorCode::InvalidFormat,
                         fmt::format("Unknown GGUF type: {}", static_cast<uint32_t>(type)));
    }
}

Result<void> GGUFFile::parse_metadata() {
    size_t offset = 24;  // After header

    for (uint64_t i = 0; i < header_.metadata_kv_count; i++) {
        // Read key
        std::string key = read_string(offset);

        // Read value type
        uint32_t value_type;
        memcpy(&value_type, static_cast<const uint8_t*>(data_) + offset, 4);
        offset += 4;

        // Read value
        auto value_result = read_value(static_cast<GGUFType>(value_type), offset);
        if (!value_result.ok()) {
            return value_result.error();
        }

        metadata_[key] = std::move(value_result).take();
    }

    return {};
}

Result<void> GGUFFile::parse_tensors() {
    // Calculate where metadata ended
    size_t offset = 24;  // After header

    // Skip through metadata to find tensor info start
    for (uint64_t i = 0; i < header_.metadata_kv_count; i++) {
        // Skip key
        uint64_t key_len;
        memcpy(&key_len, static_cast<const uint8_t*>(data_) + offset, 8);
        offset += 8 + key_len;

        // Read and skip value type
        uint32_t value_type;
        memcpy(&value_type, static_cast<const uint8_t*>(data_) + offset, 4);
        offset += 4;

        // Skip value (need to parse to get size)
        auto value_result = read_value(static_cast<GGUFType>(value_type), offset);
        if (!value_result.ok()) {
            return value_result.error();
        }
        (void)value_result;  // We just needed to advance offset
    }

    // Now parse tensor infos
    tensors_.reserve(header_.tensor_count);

    for (uint64_t i = 0; i < header_.tensor_count; i++) {
        GGUFTensorInfo info;

        // Read name
        info.name = read_string(offset);

        // Read dimensions count
        memcpy(&info.n_dims, static_cast<const uint8_t*>(data_) + offset, 4);
        offset += 4;

        // Read dimensions
        info.dimensions.resize(info.n_dims);
        for (uint32_t d = 0; d < info.n_dims; d++) {
            memcpy(&info.dimensions[d], static_cast<const uint8_t*>(data_) + offset, 8);
            offset += 8;
        }

        // Read type
        uint32_t type_val;
        memcpy(&type_val, static_cast<const uint8_t*>(data_) + offset, 4);
        info.type = static_cast<GGMLType>(type_val);
        offset += 4;

        // Read offset
        memcpy(&info.offset, static_cast<const uint8_t*>(data_) + offset, 8);
        offset += 8;

        tensors_.push_back(info);
    }

    // Align to 32 bytes for data section
    data_offset_ = (offset + 31) & ~static_cast<size_t>(31);

    return {};
}

std::vector<std::string> GGUFFile::metadata_keys() const {
    std::vector<std::string> keys;
    keys.reserve(metadata_.size());
    for (const auto& [k, v] : metadata_) {
        keys.push_back(k);
    }
    return keys;
}

bool GGUFFile::has_metadata(const std::string& key) const {
    return metadata_.find(key) != metadata_.end();
}

std::optional<GGUFValue> GGUFFile::get_metadata(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<std::string> GGUFFile::get_architecture() const {
    return get_metadata_as<std::string>("general.architecture");
}

std::optional<uint32_t> GGUFFile::get_context_length() const {
    auto arch = get_architecture();
    if (!arch) return std::nullopt;
    return get_metadata_as<uint32_t>(*arch + ".context_length");
}

std::optional<uint32_t> GGUFFile::get_embedding_length() const {
    auto arch = get_architecture();
    if (!arch) return std::nullopt;
    return get_metadata_as<uint32_t>(*arch + ".embedding_length");
}

std::optional<uint32_t> GGUFFile::get_block_count() const {
    auto arch = get_architecture();
    if (!arch) return std::nullopt;
    return get_metadata_as<uint32_t>(*arch + ".block_count");
}

std::optional<uint32_t> GGUFFile::get_head_count() const {
    auto arch = get_architecture();
    if (!arch) return std::nullopt;
    return get_metadata_as<uint32_t>(*arch + ".attention.head_count");
}

std::optional<uint32_t> GGUFFile::get_head_count_kv() const {
    auto arch = get_architecture();
    if (!arch) return std::nullopt;
    return get_metadata_as<uint32_t>(*arch + ".attention.head_count_kv");
}

const GGUFTensorInfo* GGUFFile::find_tensor(const std::string& name) const {
    for (const auto& t : tensors_) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

const void* GGUFFile::tensor_data(const GGUFTensorInfo& info) const {
    return static_cast<const uint8_t*>(data_) + data_offset_ + info.offset;
}

std::string GGUFFile::summary() const {
    std::ostringstream ss;
    ss << "GGUF File: " << path_ << "\n";
    ss << "  Version: " << header_.version << "\n";
    ss << "  File size: " << file_size_ << " bytes\n";
    ss << "  Metadata entries: " << metadata_.size() << "\n";
    ss << "  Tensors: " << tensors_.size() << "\n";

    if (auto arch = get_architecture()) {
        ss << "  Architecture: " << *arch << "\n";
    }
    if (auto ctx = get_context_length()) {
        ss << "  Context length: " << *ctx << "\n";
    }
    if (auto emb = get_embedding_length()) {
        ss << "  Embedding length: " << *emb << "\n";
    }
    if (auto blocks = get_block_count()) {
        ss << "  Layers: " << *blocks << "\n";
    }

    // Summarize tensor types
    std::unordered_map<GGMLType, int> type_counts;
    size_t total_size = 0;
    for (const auto& t : tensors_) {
        type_counts[t.type]++;
        total_size += t.size_bytes();
    }

    ss << "  Tensor types:\n";
    for (const auto& [type, count] : type_counts) {
        ss << "    " << ggml_type_name(type) << ": " << count << "\n";
    }
    ss << "  Total tensor data: " << total_size / (1024 * 1024) << " MB\n";

    return ss.str();
}

// =============================================================================
// FP16/FP32 Conversion Helpers
// =============================================================================

namespace {

// Convert FP16 to FP32
inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            uint32_t bits = sign << 31;
            float result;
            memcpy(&result, &bits, 4);
            return result;
        }
        // Denormal - convert to normalized
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        // Inf or NaN
        uint32_t bits = (sign << 31) | 0x7F800000 | (mant << 13);
        float result;
        memcpy(&result, &bits, 4);
        return result;
    }

    uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &bits, 4);
    return result;
}

// Convert FP32 to FP16
inline uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;

    if (exp <= 0) {
        return static_cast<uint16_t>(sign);  // Underflow to zero
    } else if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00);  // Overflow to inf
    }
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

}  // anonymous namespace

// =============================================================================
// ModelLoader Implementation
// =============================================================================

ModelLoader::ModelLoader(IComputeBackend* backend)
    : backend_(backend)
{
}

Result<ModelInfo> ModelLoader::load_info(const std::string& path) {
    auto file_result = GGUFFile::open(path);
    if (!file_result.ok()) {
        return file_result.error();
    }

    auto& file = file_result.value();
    ModelInfo info;

    if (auto arch = file.get_architecture()) {
        info.architecture = *arch;
    }

    if (auto name = file.get_metadata_as<std::string>("general.name")) {
        info.name = *name;
    }

    if (auto ctx = file.get_context_length()) {
        info.context_length = *ctx;
    }

    if (auto emb = file.get_embedding_length()) {
        info.embedding_length = *emb;
    }

    if (auto blocks = file.get_block_count()) {
        info.block_count = *blocks;
    }

    if (auto heads = file.get_head_count()) {
        info.head_count = *heads;
    }

    if (auto heads_kv = file.get_head_count_kv()) {
        info.head_count_kv = *heads_kv;
    }

    // Determine predominant weight type
    std::unordered_map<GGMLType, size_t> type_sizes;
    for (const auto& t : file.tensors()) {
        type_sizes[t.type] += t.size_bytes();
    }

    size_t max_size = 0;
    for (const auto& [type, size] : type_sizes) {
        if (size > max_size) {
            max_size = size;
            info.weight_type = type;
        }
    }

    return info;
}

Result<std::unordered_map<std::string, Tensor>> ModelLoader::load_weights(
    const GGUFFile& file)
{
    std::unordered_map<std::string, Tensor> tensors;

    for (const auto& info : file.tensors()) {
        auto tensor_result = load_tensor(file, info.name);
        if (!tensor_result.ok()) {
            GRANITE_LOG_WARN("Failed to load tensor '{}': {}",
                            info.name, tensor_result.error().message());
            continue;
        }

        tensors[info.name] = std::move(tensor_result).take();
    }

    return tensors;
}

Result<Tensor> ModelLoader::load_tensor(
    const GGUFFile& file,
    const std::string& name)
{
    const GGUFTensorInfo* info = file.find_tensor(name);
    if (!info) {
        GRANITE_FAIL(ErrorCode::InvalidArgument,
                     fmt::format("Tensor not found: {}", name));
    }

    const void* data = file.tensor_data(*info);

    // Convert GGUF dimensions to Granite shape (reversed order)
    std::vector<int64_t> shape(info->dimensions.rbegin(), info->dimensions.rend());

    // Check if we need to dequantize
    auto dtype = ggml_type_to_dtype(info->type);
    if (dtype) {
        // Direct copy for non-quantized types
        auto tensor_result = Tensor::allocate(shape, *dtype, backend_);
        if (!tensor_result.ok()) {
            return tensor_result.error();
        }

        auto tensor = std::move(tensor_result).take();

        // Copy data to tensor
        auto write_result = backend_->write_buffer(
            tensor.buffer(), data, info->size_bytes());

        if (!write_result.ok()) {
            return write_result.error();
        }

        return tensor;
    } else {
        // Need to dequantize
        return dequantize_tensor(*info, data);
    }
}

Result<Tensor> ModelLoader::dequantize_tensor(
    const GGUFTensorInfo& info,
    const void* data)
{
    // Convert shape
    std::vector<int64_t> shape(info.dimensions.rbegin(), info.dimensions.rend());

    // Allocate FP16 tensor for output
    auto tensor_result = Tensor::allocate(shape, DataType::FP16, backend_);
    if (!tensor_result.ok()) {
        return tensor_result.error();
    }

    auto tensor = std::move(tensor_result).take();

    // Map buffer for writing
    auto map_result = backend_->map_buffer(tensor.buffer());
    if (!map_result.ok()) {
        return map_result.error();
    }

    auto* output = static_cast<uint16_t*>(map_result.value());  // FP16 as uint16
    size_t numel = info.numel();

    // Dequantize based on type
    switch (info.type) {
        case GGMLType::Q8_0: {
            // Q8_0: 32 elements per block, 2 byte scale + 32 byte data
            const uint8_t* src = static_cast<const uint8_t*>(data);
            size_t num_blocks = (numel + 31) / 32;

            for (size_t b = 0; b < num_blocks; b++) {
                uint16_t scale_bits;
                memcpy(&scale_bits, src, 2);
                src += 2;

                float scale = fp16_to_fp32(scale_bits);

                for (int i = 0; i < 32 && (b * 32 + i) < numel; i++) {
                    int8_t q = static_cast<int8_t>(src[i]);
                    float f = static_cast<float>(q) * scale;
                    output[b * 32 + i] = fp32_to_fp16(f);
                }
                src += 32;
            }
            break;
        }

        case GGMLType::Q4_0: {
            // Q4_0: 32 elements per block, 2 byte scale + 16 byte data (4 bits each)
            const uint8_t* src = static_cast<const uint8_t*>(data);
            size_t num_blocks = (numel + 31) / 32;

            for (size_t b = 0; b < num_blocks; b++) {
                uint16_t scale_bits;
                memcpy(&scale_bits, src, 2);
                src += 2;

                float scale = fp16_to_fp32(scale_bits);

                for (int i = 0; i < 16 && (b * 32 + i * 2) < numel; i++) {
                    uint8_t byte = src[i];
                    int8_t low = (byte & 0xF) - 8;   // Q4 is stored with bias of 8
                    int8_t high = (byte >> 4) - 8;

                    if ((b * 32 + i * 2) < numel) {
                        output[b * 32 + i * 2] = fp32_to_fp16(static_cast<float>(low) * scale);
                    }
                    if ((b * 32 + i * 2 + 1) < numel) {
                        output[b * 32 + i * 2 + 1] = fp32_to_fp16(static_cast<float>(high) * scale);
                    }
                }
                src += 16;
            }
            break;
        }

        case GGMLType::Q4_K: {
            // Q4_K: 256 elements per super-block, 144 bytes per block
            // Structure (matching GGML's block_q4_K):
            //   - d: fp16 scale (2 bytes)
            //   - dmin: fp16 min (2 bytes)
            //   - scales: 12 bytes (8 6-bit scales + 8 6-bit mins, packed)
            //   - qs: 128 bytes (4-bit quantized values)
            const uint8_t* src = static_cast<const uint8_t*>(data);
            size_t num_blocks = (numel + 255) / 256;

            // Lambda to match GGML's get_scale_min_k4
            auto get_scale_min_k4 = [](int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
                if (j < 4) {
                    sc = q[j] & 63;
                    m = q[j + 4] & 63;
                } else {
                    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
                }
            };

            for (size_t b = 0; b < num_blocks; b++) {
                // Read global scales
                uint16_t d_bits, dmin_bits;
                memcpy(&d_bits, src, 2);
                memcpy(&dmin_bits, src + 2, 2);
                float d = fp16_to_fp32(d_bits);
                float dmin = fp16_to_fp32(dmin_bits);

                const uint8_t* scales_ptr = src + 4;
                const uint8_t* qs = src + 16;  // After d, dmin, scales (4 + 12 = 16)

                int is = 0;
                // Process 256 elements in groups of 64 (4 iterations)
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m;

                    // Get scale/min for first 32 elements (low nibble)
                    get_scale_min_k4(is, scales_ptr, sc, m);
                    float d1 = d * static_cast<float>(sc);
                    float m1 = dmin * static_cast<float>(m);

                    // Get scale/min for next 32 elements (high nibble)
                    get_scale_min_k4(is + 1, scales_ptr, sc, m);
                    float d2 = d * static_cast<float>(sc);
                    float m2 = dmin * static_cast<float>(m);

                    // Dequantize 32 elements using low nibble
                    for (int l = 0; l < 32; l++) {
                        size_t idx = b * 256 + j + l;
                        if (idx >= numel) break;
                        float val = d1 * static_cast<float>(qs[l] & 0xF) - m1;
                        output[idx] = fp32_to_fp16(val);
                    }

                    // Dequantize 32 elements using high nibble
                    for (int l = 0; l < 32; l++) {
                        size_t idx = b * 256 + j + 32 + l;
                        if (idx >= numel) break;
                        float val = d2 * static_cast<float>(qs[l] >> 4) - m2;
                        output[idx] = fp32_to_fp16(val);
                    }

                    qs += 32;
                    is += 2;
                }

                src += 144;  // Move to next block
            }
            break;
        }


        case GGMLType::Q5_K: {
            // Q5_K: 256 elements per super-block, 176 bytes per block
            // Structure (matching GGML's block_q5_K):
            //   - d: fp16 scale (2 bytes)
            //   - dmin: fp16 min (2 bytes)
            //   - scales: 12 bytes (6-bit scales and mins, packed)
            //   - qh: 32 bytes (high bits, 1 per element)
            //   - qs: 128 bytes (low 4 bits, 2 per byte)
            const uint8_t* src = static_cast<const uint8_t*>(data);
            size_t num_blocks = (numel + 255) / 256;

            // Lambda to match GGML's get_scale_min_k4
            auto get_scale_min_k4 = [](int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
                if (j < 4) {
                    sc = q[j] & 63;
                    m = q[j + 4] & 63;
                } else {
                    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
                }
            };

            for (size_t b = 0; b < num_blocks; b++) {
                // Read global scales
                uint16_t d_bits, dmin_bits;
                memcpy(&d_bits, src, 2);
                memcpy(&dmin_bits, src + 2, 2);
                float d = fp16_to_fp32(d_bits);
                float dmin = fp16_to_fp32(dmin_bits);

                const uint8_t* scales_ptr = src + 4;    // 12 bytes
                const uint8_t* qh = src + 16;           // 32 bytes (high bits)
                const uint8_t* qs = src + 48;           // 128 bytes (low 4 bits)

                int is = 0;
                // Process 256 elements in groups of 64 (4 iterations)
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m;

                    // Get scale/min for first 32 elements (low nibble)
                    get_scale_min_k4(is, scales_ptr, sc, m);
                    float d1 = d * static_cast<float>(sc);
                    float m1 = dmin * static_cast<float>(m);

                    // Get scale/min for next 32 elements (high nibble)
                    get_scale_min_k4(is + 1, scales_ptr, sc, m);
                    float d2 = d * static_cast<float>(sc);
                    float m2 = dmin * static_cast<float>(m);

                    // Dequantize 32 elements using low nibble + high bit
                    for (int l = 0; l < 32; l++) {
                        size_t idx = b * 256 + j + l;
                        if (idx >= numel) break;
                        // Get high bit from qh
                        int qh_idx = (j / 8) + (l / 8);
                        int qh_bit = l % 8;
                        uint8_t hbit = (qh[qh_idx] >> qh_bit) & 1;
                        // Combine low 4 bits + high bit to get 5-bit value
                        int q5 = (qs[l] & 0xF) | (hbit << 4);
                        float val = d1 * static_cast<float>(q5) - m1;
                        output[idx] = fp32_to_fp16(val);
                    }

                    // Dequantize 32 elements using high nibble + high bit
                    for (int l = 0; l < 32; l++) {
                        size_t idx = b * 256 + j + 32 + l;
                        if (idx >= numel) break;
                        // Get high bit from qh (offset by 4 bytes for high nibble)
                        int qh_idx = (j / 8) + 4 + (l / 8);
                        int qh_bit = l % 8;
                        uint8_t hbit = (qh[qh_idx] >> qh_bit) & 1;
                        // Combine high 4 bits + high bit to get 5-bit value
                        int q5 = (qs[l] >> 4) | (hbit << 4);
                        float val = d2 * static_cast<float>(q5) - m2;
                        output[idx] = fp32_to_fp16(val);
                    }

                    qs += 32;
                    is += 2;
                }

                src += 176;  // Move to next block
            }
            break;
        }

        case GGMLType::Q6_K: {
            // Q6_K: 256 elements per super-block, 210 bytes per block
            // Structure (matching GGML's block_q6_K):
            //   - ql: 128 bytes (lower 4 bits)
            //   - qh: 64 bytes (upper 2 bits)
            //   - scales: 16 bytes (8-bit signed scales)
            //   - d: fp16 scale (2 bytes)
            const uint8_t* src = static_cast<const uint8_t*>(data);
            size_t num_blocks = (numel + 255) / 256;

            for (size_t blk = 0; blk < num_blocks; blk++) {
                const uint8_t* ql = src;           // 128 bytes
                const uint8_t* qh = src + 128;     // 64 bytes
                const int8_t* sc = reinterpret_cast<const int8_t*>(src + 192);  // 16 bytes
                uint16_t d_bits;
                memcpy(&d_bits, src + 208, 2);
                float d = fp16_to_fp32(d_bits);

                // Process 256 elements in two passes of 128
                for (int n = 0; n < 256; n += 128) {
                    for (int l = 0; l < 32; l++) {
                        int is = l / 16;  // Scale index base

                        // Extract 4 6-bit values
                        int8_t q1 = static_cast<int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int8_t q3 = static_cast<int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                        size_t base_idx = blk * 256 + n;
                        if (base_idx + l < numel)
                            output[base_idx + l + 0] = fp32_to_fp16(d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1));
                        if (base_idx + l + 32 < numel)
                            output[base_idx + l + 32] = fp32_to_fp16(d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2));
                        if (base_idx + l + 64 < numel)
                            output[base_idx + l + 64] = fp32_to_fp16(d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3));
                        if (base_idx + l + 96 < numel)
                            output[base_idx + l + 96] = fp32_to_fp16(d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4));
                    }
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }

                src += 210;
            }
            break;
        }

        case GGMLType::IQ4_NL: {
            // IQ4_NL: 32 elements per block, 18 bytes per block
            // Structure: half d (2 bytes) + uint8_t qs[16] (16 bytes)
            // Uses non-linear lookup table for dequantization
            static const int8_t kvalues_iq4nl[16] = {
                -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
            };

            const uint8_t* src = static_cast<const uint8_t*>(data);
            size_t num_blocks = (numel + 31) / 32;

            for (size_t blk = 0; blk < num_blocks; blk++) {
                uint16_t d_bits;
                memcpy(&d_bits, src, 2);
                float d = fp16_to_fp32(d_bits);
                const uint8_t* qs = src + 2;

                for (int i = 0; i < 16; i++) {
                    int q0 = qs[i] & 0xF;       // Low 4 bits
                    int q1 = qs[i] >> 4;        // High 4 bits

                    size_t idx0 = blk * 32 + 2 * i;
                    size_t idx1 = blk * 32 + 2 * i + 1;

                    if (idx0 < numel)
                        output[idx0] = fp32_to_fp16(d * static_cast<float>(kvalues_iq4nl[q0]));
                    if (idx1 < numel)
                        output[idx1] = fp32_to_fp16(d * static_cast<float>(kvalues_iq4nl[q1]));
                }

                src += 18;  // Move to next block
            }
            break;
        }

        case GGMLType::IQ4_XS: {
            // IQ4_XS: 256 elements per super-block, 136 bytes per block
            // Structure: half d (2) + uint16_t scales_h (2) + uint8_t scales_l[4] (4) + uint8_t qs[128] (128)
            // Uses non-linear lookup table with per-sub-block 6-bit scales
            static const int8_t kvalues_iq4nl[16] = {
                -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
            };

            const uint8_t* src = static_cast<const uint8_t*>(data);
            size_t num_blocks = (numel + 255) / 256;

            for (size_t blk = 0; blk < num_blocks; blk++) {
                // Read d (super-block scale)
                uint16_t d_bits;
                memcpy(&d_bits, src, 2);
                float d = fp16_to_fp32(d_bits);

                // Read scales_h (high 2 bits of scales)
                uint16_t scales_h;
                memcpy(&scales_h, src + 2, 2);

                // scales_l is at src + 4 (4 bytes)
                const uint8_t* scales_l = src + 4;

                // qs is at src + 8 (128 bytes)
                const uint8_t* qs = src + 8;

                // Process 8 sub-blocks of 32 elements each
                for (int ib32 = 0; ib32 < 8; ib32++) {
                    // Extract 6-bit scale for this sub-block
                    int ls = ((scales_l[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf) |
                             (((scales_h >> (2 * ib32)) & 3) << 4);
                    float scale = d * static_cast<float>(ls - 32);

                    // Dequantize 32 elements (16 bytes)
                    for (int i = 0; i < 16; i++) {
                        int q0 = qs[ib32 * 16 + i] & 0xF;
                        int q1 = qs[ib32 * 16 + i] >> 4;

                        size_t idx0 = blk * 256 + ib32 * 32 + 2 * i;
                        size_t idx1 = blk * 256 + ib32 * 32 + 2 * i + 1;

                        if (idx0 < numel)
                            output[idx0] = fp32_to_fp16(scale * static_cast<float>(kvalues_iq4nl[q0]));
                        if (idx1 < numel)
                            output[idx1] = fp32_to_fp16(scale * static_cast<float>(kvalues_iq4nl[q1]));
                    }
                }

                src += 136;  // Move to next block
            }
            break;
        }

        case GGMLType::IQ3_S: {
            // IQ3_S: 256 elements per super-block, 110 bytes per block
            // Structure: half d (2) + uint8_t qs[64] + uint8_t qh[8] + uint8_t signs[32] + uint8_t scales[4]
            // Uses iq3s_grid[512] lookup table for dequantization
            static const uint32_t iq3s_grid[512] = {
                0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
                0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
                0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
                0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
                0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
                0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
                0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
                0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
                0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
                0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
                0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
                0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
                0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
                0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
                0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
                0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
                0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
                0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
                0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
                0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
                0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
                0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
                0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
                0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
                0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
                0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
                0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
                0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
                0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
                0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
                0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
                0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
                0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
                0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
                0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
                0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
                0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
                0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
                0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
                0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
                0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
                0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
                0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
                0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
                0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
                0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
                0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
                0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
                0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
                0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
                0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
                0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
                0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
                0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
                0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
                0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
                0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
                0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
                0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
                0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
                0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
                0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
                0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
                0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
            };
            static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

            const uint8_t* src = static_cast<const uint8_t*>(data);
            size_t num_blocks = (numel + 255) / 256;

            for (size_t blk = 0; blk < num_blocks; blk++) {
                // Read d (super-block scale)
                uint16_t d_bits;
                memcpy(&d_bits, src, 2);
                float d = fp16_to_fp32(d_bits);

                // qs is at src + 2 (64 bytes)
                const uint8_t* qs = src + 2;
                // qh is at src + 66 (8 bytes)
                const uint8_t* qh = src + 66;
                // signs is at src + 74 (32 bytes)
                const uint8_t* signs = src + 74;
                // scales is at src + 106 (4 bytes)
                const uint8_t* scales = src + 106;

                // Process 8 sub-blocks of 32 elements each
                for (int ib32 = 0; ib32 < 8; ib32++) {
                    // Extract 4-bit scale for this sub-block: 1 + 2 * scale_value
                    int scale_val = (scales[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf;
                    float dl = d * static_cast<float>(1 + 2 * scale_val);

                    // Process 2 halves of 16 elements each (il = 0, 1)
                    for (int il = 0; il < 2; il++) {
                        const uint8_t* qs_ptr = qs + 8 * ib32 + 4 * il;
                        const uint8_t* signs_ptr = signs + 4 * ib32 + 2 * il;
                        uint8_t qh_byte = qh[ib32] >> (4 * il);

                        // Process 4 qs bytes -> 16 elements (4 grids of 4 elements)
                        for (int j = 0; j < 4; j++) {
                            // Build 9-bit grid index
                            int grid_idx = qs_ptr[j] | (((qh_byte >> (3 - j)) & 1) << 8);
                            uint32_t grid_val = iq3s_grid[grid_idx];
                            const uint8_t* grid = reinterpret_cast<const uint8_t*>(&grid_val);
                            uint8_t sign_byte = signs_ptr[j / 2];

                            // 4 values from this grid entry
                            for (int k = 0; k < 4; k++) {
                                size_t idx = blk * 256 + ib32 * 32 + il * 16 + j * 4 + k;
                                if (idx < numel) {
                                    int sign_bit_idx = (j % 2) * 4 + k;
                                    float sign = (sign_byte & kmask_iq2xs[sign_bit_idx]) ? -1.0f : 1.0f;
                                    output[idx] = fp32_to_fp16(dl * static_cast<float>(grid[k]) * sign);
                                }
                            }
                        }
                    }
                }

                src += 110;  // Move to next block
            }
            break;
        }

        case GGMLType::Q2_K: {
            // Q2_K: 256 elements per super-block, 84 bytes per block
            // Structure (matching GGML's block_q2_K):
            //   - scales[16]: 4-bit scales (low nibble) and mins (high nibble)
            //   - qs[64]: 2-bit quants (4 per byte)
            //   - d: fp16 scale (2 bytes)
            //   - dmin: fp16 min (2 bytes)
            // Dequantization: w = d * (scale & 0xF) * q2 - dmin * (scale >> 4)
            const uint8_t* src = static_cast<const uint8_t*>(data);
            size_t num_blocks = (numel + 255) / 256;

            for (size_t b = 0; b < num_blocks; b++) {
                const uint8_t* scales = src;           // 16 bytes
                const uint8_t* qs = src + 16;          // 64 bytes
                uint16_t d_bits, dmin_bits;
                memcpy(&d_bits, src + 80, 2);
                memcpy(&dmin_bits, src + 82, 2);
                float d_all = fp16_to_fp32(d_bits);
                float dmin = fp16_to_fp32(dmin_bits);

                // Process 16 sub-blocks of 16 elements each
                for (int is = 0; is < 16; is++) {
                    // Each scale byte contains scale (low nibble) and min (high nibble)
                    uint8_t scale_byte = scales[is];
                    float dl = d_all * static_cast<float>(scale_byte & 0xF);
                    float ml = dmin * static_cast<float>(scale_byte >> 4);

                    // 16 elements from 4 bytes of qs (4 per byte)
                    for (int j = 0; j < 4; j++) {
                        uint8_t q_byte = qs[is * 4 + j];
                        for (int k = 0; k < 4; k++) {
                            size_t idx = b * 256 + is * 16 + j * 4 + k;
                            if (idx < numel) {
                                // Extract 2-bit quant
                                int q2 = (q_byte >> (k * 2)) & 3;
                                float val = dl * static_cast<float>(q2) - ml;
                                output[idx] = fp32_to_fp16(val);
                            }
                        }
                    }
                }

                src += 84;  // Move to next block
            }
            break;
        }

        case GGMLType::Q3_K: {
            // Q3_K: 256 elements per super-block, 110 bytes per block
            // Structure (matching GGML's block_q3_K):
            //   - hmask: 32 bytes (high bit of 3-bit quants)
            //   - qs: 64 bytes (low 2 bits of 3-bit quants)
            //   - scales: 12 bytes (6-bit scales, packed)
            //   - d: fp16 scale (2 bytes)
            const uint8_t* src = static_cast<const uint8_t*>(data);
            size_t num_blocks = (numel + 255) / 256;

            const uint32_t kmask1 = 0x03030303;
            const uint32_t kmask2 = 0x0f0f0f0f;

            for (size_t b = 0; b < num_blocks; b++) {
                const uint8_t* hmask = src;           // 32 bytes
                const uint8_t* qs = src + 32;         // 64 bytes
                const uint8_t* scales_raw = src + 96; // 12 bytes
                uint16_t d_bits;
                memcpy(&d_bits, src + 108, 2);
                float d_all = fp16_to_fp32(d_bits);

                // Unpack 16 6-bit scales from 12 bytes (matches llama.cpp)
                uint32_t aux[4];
                memcpy(aux, scales_raw, 12);
                uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

                const uint8_t* q = qs;
                const uint8_t* hm = hmask;
                uint8_t m = 1;
                int is = 0;

                // Process 256 elements in two passes of 128
                for (int n = 0; n < 256; n += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j) {
                        float dl = d_all * (scales[is++] - 32);

                        // First 16 elements
                        for (int l = 0; l < 16; ++l) {
                            size_t idx = b * 256 + n + j * 32 + l;
                            if (idx < numel) {
                                int8_t q3 = ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                                output[idx] = fp32_to_fp16(dl * static_cast<float>(q3));
                            }
                        }

                        dl = d_all * (scales[is++] - 32);

                        // Next 16 elements
                        for (int l = 0; l < 16; ++l) {
                            size_t idx = b * 256 + n + j * 32 + 16 + l;
                            if (idx < numel) {
                                int8_t q3 = ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                                output[idx] = fp32_to_fp16(dl * static_cast<float>(q3));
                            }
                        }

                        shift += 2;
                        m <<= 1;
                    }
                    q += 32;
                }

                src += 110;  // Move to next block
            }
            break;
        }

        default:
            backend_->unmap_buffer(tensor.buffer());
            GRANITE_FAIL(ErrorCode::NotImplemented,
                         fmt::format("Dequantization not implemented for type: {}",
                                     ggml_type_name(info.type)));
    }

    backend_->unmap_buffer(tensor.buffer());
    return tensor;
}

}  // namespace granite

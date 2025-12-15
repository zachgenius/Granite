#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <granite/granite.h>

#include <cstring>
#include <fstream>
#include <filesystem>

using namespace granite;

// =============================================================================
// GGML Type Utilities Tests
// =============================================================================

TEST_CASE("GGML type names", "[gguf]") {
    REQUIRE(std::string(ggml_type_name(GGMLType::F32)) == "F32");
    REQUIRE(std::string(ggml_type_name(GGMLType::F16)) == "F16");
    REQUIRE(std::string(ggml_type_name(GGMLType::BF16)) == "BF16");
    REQUIRE(std::string(ggml_type_name(GGMLType::Q4_0)) == "Q4_0");
    REQUIRE(std::string(ggml_type_name(GGMLType::Q4_1)) == "Q4_1");
    REQUIRE(std::string(ggml_type_name(GGMLType::Q5_0)) == "Q5_0");
    REQUIRE(std::string(ggml_type_name(GGMLType::Q5_1)) == "Q5_1");
    REQUIRE(std::string(ggml_type_name(GGMLType::Q8_0)) == "Q8_0");
    REQUIRE(std::string(ggml_type_name(GGMLType::Q8_1)) == "Q8_1");
    REQUIRE(std::string(ggml_type_name(GGMLType::Q2_K)) == "Q2_K");
    REQUIRE(std::string(ggml_type_name(GGMLType::Q3_K)) == "Q3_K");
    REQUIRE(std::string(ggml_type_name(GGMLType::Q4_K)) == "Q4_K");
    REQUIRE(std::string(ggml_type_name(GGMLType::Q5_K)) == "Q5_K");
    REQUIRE(std::string(ggml_type_name(GGMLType::Q6_K)) == "Q6_K");
    REQUIRE(std::string(ggml_type_name(GGMLType::Q8_K)) == "Q8_K");
    REQUIRE(std::string(ggml_type_name(GGMLType::I8)) == "I8");
    REQUIRE(std::string(ggml_type_name(GGMLType::I16)) == "I16");
    REQUIRE(std::string(ggml_type_name(GGMLType::I32)) == "I32");
    REQUIRE(std::string(ggml_type_name(GGMLType::I64)) == "I64");
    REQUIRE(std::string(ggml_type_name(GGMLType::F64)) == "F64");
}

TEST_CASE("GGML type sizes", "[gguf]") {
    SECTION("non-quantized types") {
        REQUIRE(ggml_type_size(GGMLType::F32) == 4);
        REQUIRE(ggml_type_size(GGMLType::F16) == 2);
        REQUIRE(ggml_type_size(GGMLType::BF16) == 2);
        REQUIRE(ggml_type_size(GGMLType::I8) == 1);
        REQUIRE(ggml_type_size(GGMLType::I16) == 2);
        REQUIRE(ggml_type_size(GGMLType::I32) == 4);
        REQUIRE(ggml_type_size(GGMLType::I64) == 8);
        REQUIRE(ggml_type_size(GGMLType::F64) == 8);
    }

    SECTION("quantized types return 0") {
        REQUIRE(ggml_type_size(GGMLType::Q4_0) == 0);
        REQUIRE(ggml_type_size(GGMLType::Q8_0) == 0);
        REQUIRE(ggml_type_size(GGMLType::Q4_K) == 0);
    }
}

TEST_CASE("GGML block sizes", "[gguf]") {
    SECTION("non-quantized have block size 1") {
        REQUIRE(ggml_type_block_size(GGMLType::F32) == 1);
        REQUIRE(ggml_type_block_size(GGMLType::F16) == 1);
        REQUIRE(ggml_type_block_size(GGMLType::BF16) == 1);
        REQUIRE(ggml_type_block_size(GGMLType::I8) == 1);
        REQUIRE(ggml_type_block_size(GGMLType::I32) == 1);
    }

    SECTION("standard quantized have 32 element blocks") {
        REQUIRE(ggml_type_block_size(GGMLType::Q4_0) == 32);
        REQUIRE(ggml_type_block_size(GGMLType::Q4_1) == 32);
        REQUIRE(ggml_type_block_size(GGMLType::Q5_0) == 32);
        REQUIRE(ggml_type_block_size(GGMLType::Q5_1) == 32);
        REQUIRE(ggml_type_block_size(GGMLType::Q8_0) == 32);
        REQUIRE(ggml_type_block_size(GGMLType::Q8_1) == 32);
    }

    SECTION("K-quants have 256 element blocks") {
        REQUIRE(ggml_type_block_size(GGMLType::Q2_K) == 256);
        REQUIRE(ggml_type_block_size(GGMLType::Q3_K) == 256);
        REQUIRE(ggml_type_block_size(GGMLType::Q4_K) == 256);
        REQUIRE(ggml_type_block_size(GGMLType::Q5_K) == 256);
        REQUIRE(ggml_type_block_size(GGMLType::Q6_K) == 256);
        REQUIRE(ggml_type_block_size(GGMLType::Q8_K) == 256);
    }
}

TEST_CASE("GGML bytes per block", "[gguf]") {
    SECTION("non-quantized") {
        REQUIRE(ggml_type_bytes_per_block(GGMLType::F32) == 4);
        REQUIRE(ggml_type_bytes_per_block(GGMLType::F16) == 2);
        REQUIRE(ggml_type_bytes_per_block(GGMLType::I8) == 1);
    }

    SECTION("standard quantized") {
        // Q4_0: 32 elements, 2 bytes scale + 16 bytes data = 18 bytes
        REQUIRE(ggml_type_bytes_per_block(GGMLType::Q4_0) == 18);
        // Q4_1: 32 elements, 2 bytes scale + 2 bytes min + 16 bytes data = 20 bytes
        REQUIRE(ggml_type_bytes_per_block(GGMLType::Q4_1) == 20);
        // Q8_0: 32 elements, 2 bytes scale + 32 bytes data = 34 bytes
        REQUIRE(ggml_type_bytes_per_block(GGMLType::Q8_0) == 34);
        // Q8_1: 32 elements, 4 bytes scale + 4 bytes sum + 32 bytes data = 40 bytes
        REQUIRE(ggml_type_bytes_per_block(GGMLType::Q8_1) == 40);
    }

    SECTION("K-quants") {
        REQUIRE(ggml_type_bytes_per_block(GGMLType::Q2_K) == 84);
        REQUIRE(ggml_type_bytes_per_block(GGMLType::Q3_K) == 110);
        REQUIRE(ggml_type_bytes_per_block(GGMLType::Q4_K) == 144);
        REQUIRE(ggml_type_bytes_per_block(GGMLType::Q5_K) == 176);
        REQUIRE(ggml_type_bytes_per_block(GGMLType::Q6_K) == 210);
        REQUIRE(ggml_type_bytes_per_block(GGMLType::Q8_K) == 292);
    }
}

TEST_CASE("GGML type is quantized", "[gguf]") {
    SECTION("non-quantized types") {
        REQUIRE(ggml_type_is_quantized(GGMLType::F32) == false);
        REQUIRE(ggml_type_is_quantized(GGMLType::F16) == false);
        REQUIRE(ggml_type_is_quantized(GGMLType::BF16) == false);
        REQUIRE(ggml_type_is_quantized(GGMLType::I8) == false);
        REQUIRE(ggml_type_is_quantized(GGMLType::I16) == false);
        REQUIRE(ggml_type_is_quantized(GGMLType::I32) == false);
        REQUIRE(ggml_type_is_quantized(GGMLType::I64) == false);
        REQUIRE(ggml_type_is_quantized(GGMLType::F64) == false);
    }

    SECTION("quantized types") {
        REQUIRE(ggml_type_is_quantized(GGMLType::Q4_0) == true);
        REQUIRE(ggml_type_is_quantized(GGMLType::Q4_1) == true);
        REQUIRE(ggml_type_is_quantized(GGMLType::Q5_0) == true);
        REQUIRE(ggml_type_is_quantized(GGMLType::Q5_1) == true);
        REQUIRE(ggml_type_is_quantized(GGMLType::Q8_0) == true);
        REQUIRE(ggml_type_is_quantized(GGMLType::Q8_1) == true);
        REQUIRE(ggml_type_is_quantized(GGMLType::Q4_K) == true);
        REQUIRE(ggml_type_is_quantized(GGMLType::Q6_K) == true);
    }
}

TEST_CASE("GGML type to DataType conversion", "[gguf]") {
    SECTION("convertible types") {
        REQUIRE(ggml_type_to_dtype(GGMLType::F32) == DataType::FP32);
        REQUIRE(ggml_type_to_dtype(GGMLType::F16) == DataType::FP16);
        REQUIRE(ggml_type_to_dtype(GGMLType::BF16) == DataType::BF16);
        REQUIRE(ggml_type_to_dtype(GGMLType::I8) == DataType::INT8);
        REQUIRE(ggml_type_to_dtype(GGMLType::I16) == DataType::INT16);
        REQUIRE(ggml_type_to_dtype(GGMLType::I32) == DataType::INT32);
    }

    SECTION("quantized types return nullopt") {
        REQUIRE(ggml_type_to_dtype(GGMLType::Q4_0) == std::nullopt);
        REQUIRE(ggml_type_to_dtype(GGMLType::Q8_0) == std::nullopt);
        REQUIRE(ggml_type_to_dtype(GGMLType::Q4_K) == std::nullopt);
    }
}

// =============================================================================
// GGUFTensorInfo Tests
// =============================================================================

TEST_CASE("GGUFTensorInfo numel", "[gguf]") {
    GGUFTensorInfo info;

    SECTION("empty dimensions is scalar") {
        info.dimensions = {};
        REQUIRE(info.numel() == 1);
    }

    SECTION("1D tensor") {
        info.dimensions = {128};
        REQUIRE(info.numel() == 128);
    }

    SECTION("2D tensor") {
        info.dimensions = {32, 64};
        REQUIRE(info.numel() == 32 * 64);
    }

    SECTION("3D tensor") {
        info.dimensions = {4, 8, 16};
        REQUIRE(info.numel() == 4 * 8 * 16);
    }

    SECTION("4D tensor") {
        info.dimensions = {2, 3, 4, 5};
        REQUIRE(info.numel() == 2 * 3 * 4 * 5);
    }
}

TEST_CASE("GGUFTensorInfo size_bytes", "[gguf]") {
    GGUFTensorInfo info;

    SECTION("F32 tensor") {
        info.type = GGMLType::F32;
        info.dimensions = {64, 64};  // 4096 elements
        REQUIRE(info.size_bytes() == 4096 * 4);
    }

    SECTION("F16 tensor") {
        info.type = GGMLType::F16;
        info.dimensions = {128, 256};  // 32768 elements
        REQUIRE(info.size_bytes() == 32768 * 2);
    }

    SECTION("Q8_0 tensor aligned") {
        // Q8_0: 32 elements per block, 34 bytes per block
        info.type = GGMLType::Q8_0;
        info.dimensions = {1024};  // 1024 / 32 = 32 blocks
        REQUIRE(info.size_bytes() == 32 * 34);
    }

    SECTION("Q8_0 tensor not aligned") {
        // 1000 elements -> ceil(1000/32) = 32 blocks (rounds up)
        info.type = GGMLType::Q8_0;
        info.dimensions = {1000};
        size_t num_blocks = (1000 + 31) / 32;  // 32 blocks
        REQUIRE(info.size_bytes() == num_blocks * 34);
    }

    SECTION("Q4_0 tensor") {
        // Q4_0: 32 elements per block, 18 bytes per block
        info.type = GGMLType::Q4_0;
        info.dimensions = {4096};  // 4096 / 32 = 128 blocks
        REQUIRE(info.size_bytes() == 128 * 18);
    }

    SECTION("Q4_K tensor") {
        // Q4_K: 256 elements per block, 144 bytes per block
        info.type = GGMLType::Q4_K;
        info.dimensions = {4096};  // 4096 / 256 = 16 blocks
        REQUIRE(info.size_bytes() == 16 * 144);
    }
}

// =============================================================================
// GGUFFile Tests
// =============================================================================

TEST_CASE("GGUFFile open non-existent file", "[gguf]") {
    auto result = GGUFFile::open("/nonexistent/path/to/file.gguf");
    REQUIRE_FALSE(result.ok());
    REQUIRE(result.error().code() == ErrorCode::FileNotFound);
}

TEST_CASE("GGUFFile open invalid magic", "[gguf]") {
    // Create a temp file with invalid magic
    auto temp_path = std::filesystem::temp_directory_path() / "test_invalid_magic.gguf";

    {
        std::ofstream file(temp_path, std::ios::binary);
        uint32_t invalid_magic = 0x12345678;
        file.write(reinterpret_cast<const char*>(&invalid_magic), 4);
        uint32_t version = 3;
        file.write(reinterpret_cast<const char*>(&version), 4);
        uint64_t tensor_count = 0;
        file.write(reinterpret_cast<const char*>(&tensor_count), 8);
        uint64_t metadata_count = 0;
        file.write(reinterpret_cast<const char*>(&metadata_count), 8);
    }

    auto result = GGUFFile::open(temp_path.string());
    REQUIRE_FALSE(result.ok());
    REQUIRE(result.error().code() == ErrorCode::InvalidFormat);

    std::filesystem::remove(temp_path);
}

TEST_CASE("GGUFFile open invalid version", "[gguf]") {
    auto temp_path = std::filesystem::temp_directory_path() / "test_invalid_version.gguf";

    {
        std::ofstream file(temp_path, std::ios::binary);
        uint32_t magic = GGUF_MAGIC;
        file.write(reinterpret_cast<const char*>(&magic), 4);
        uint32_t invalid_version = 99;  // Unsupported version
        file.write(reinterpret_cast<const char*>(&invalid_version), 4);
        uint64_t tensor_count = 0;
        file.write(reinterpret_cast<const char*>(&tensor_count), 8);
        uint64_t metadata_count = 0;
        file.write(reinterpret_cast<const char*>(&metadata_count), 8);
    }

    auto result = GGUFFile::open(temp_path.string());
    REQUIRE_FALSE(result.ok());
    REQUIRE(result.error().code() == ErrorCode::InvalidFormat);

    std::filesystem::remove(temp_path);
}

TEST_CASE("GGUFFile file too small", "[gguf]") {
    auto temp_path = std::filesystem::temp_directory_path() / "test_too_small.gguf";

    {
        std::ofstream file(temp_path, std::ios::binary);
        uint32_t magic = GGUF_MAGIC;
        file.write(reinterpret_cast<const char*>(&magic), 4);
        // File ends here - too small for header
    }

    auto result = GGUFFile::open(temp_path.string());
    REQUIRE_FALSE(result.ok());
    REQUIRE(result.error().code() == ErrorCode::InvalidFormat);

    std::filesystem::remove(temp_path);
}

// Helper to write a GGUF string (length-prefixed)
static void write_gguf_string(std::ostream& out, const std::string& str) {
    uint64_t len = str.size();
    out.write(reinterpret_cast<const char*>(&len), 8);
    out.write(str.data(), len);
}

TEST_CASE("GGUFFile open minimal valid file", "[gguf]") {
    auto temp_path = std::filesystem::temp_directory_path() / "test_minimal.gguf";

    {
        std::ofstream file(temp_path, std::ios::binary);

        // Header
        uint32_t magic = GGUF_MAGIC;
        file.write(reinterpret_cast<const char*>(&magic), 4);
        uint32_t version = 3;
        file.write(reinterpret_cast<const char*>(&version), 4);
        uint64_t tensor_count = 0;
        file.write(reinterpret_cast<const char*>(&tensor_count), 8);
        uint64_t metadata_count = 1;
        file.write(reinterpret_cast<const char*>(&metadata_count), 8);

        // One metadata entry: general.architecture = "test"
        write_gguf_string(file, "general.architecture");
        uint32_t value_type = static_cast<uint32_t>(GGUFType::STRING);
        file.write(reinterpret_cast<const char*>(&value_type), 4);
        write_gguf_string(file, "test");
    }

    auto result = GGUFFile::open(temp_path.string());
    REQUIRE(result.ok());

    auto& gguf = result.value();
    REQUIRE(gguf.is_open());
    REQUIRE(gguf.version() == 3);
    REQUIRE(gguf.tensor_count() == 0);
    REQUIRE(gguf.has_metadata("general.architecture"));
    REQUIRE(gguf.get_architecture() == "test");

    std::filesystem::remove(temp_path);
}

TEST_CASE("GGUFFile with tensor info", "[gguf]") {
    auto temp_path = std::filesystem::temp_directory_path() / "test_with_tensor.gguf";

    {
        std::ofstream file(temp_path, std::ios::binary);

        // Header
        uint32_t magic = GGUF_MAGIC;
        file.write(reinterpret_cast<const char*>(&magic), 4);
        uint32_t version = 3;
        file.write(reinterpret_cast<const char*>(&version), 4);
        uint64_t tensor_count = 1;
        file.write(reinterpret_cast<const char*>(&tensor_count), 8);
        uint64_t metadata_count = 0;
        file.write(reinterpret_cast<const char*>(&metadata_count), 8);

        // Tensor info
        write_gguf_string(file, "test_tensor");
        uint32_t n_dims = 2;
        file.write(reinterpret_cast<const char*>(&n_dims), 4);
        uint64_t dim0 = 64;
        file.write(reinterpret_cast<const char*>(&dim0), 8);
        uint64_t dim1 = 128;
        file.write(reinterpret_cast<const char*>(&dim1), 8);
        uint32_t type = static_cast<uint32_t>(GGMLType::F16);
        file.write(reinterpret_cast<const char*>(&type), 4);
        uint64_t offset = 0;
        file.write(reinterpret_cast<const char*>(&offset), 8);

        // Pad to 32-byte alignment for data section
        auto pos = file.tellp();
        size_t data_offset = (static_cast<size_t>(pos) + 31) & ~static_cast<size_t>(31);
        while (file.tellp() < static_cast<std::streamoff>(data_offset)) {
            char zero = 0;
            file.write(&zero, 1);
        }

        // Tensor data (dummy)
        std::vector<uint16_t> data(64 * 128, 0);
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * 2);
    }

    auto result = GGUFFile::open(temp_path.string());
    REQUIRE(result.ok());

    auto& gguf = result.value();
    REQUIRE(gguf.tensor_count() == 1);

    const auto& tensors = gguf.tensors();
    REQUIRE(tensors.size() == 1);
    REQUIRE(tensors[0].name == "test_tensor");
    REQUIRE(tensors[0].n_dims == 2);
    REQUIRE(tensors[0].dimensions.size() == 2);
    REQUIRE(tensors[0].dimensions[0] == 64);
    REQUIRE(tensors[0].dimensions[1] == 128);
    REQUIRE(tensors[0].type == GGMLType::F16);

    // Find tensor by name
    const auto* found = gguf.find_tensor("test_tensor");
    REQUIRE(found != nullptr);
    REQUIRE(found->name == "test_tensor");

    // Non-existent tensor
    REQUIRE(gguf.find_tensor("nonexistent") == nullptr);

    std::filesystem::remove(temp_path);
}

TEST_CASE("GGUFFile metadata types", "[gguf]") {
    auto temp_path = std::filesystem::temp_directory_path() / "test_metadata.gguf";

    {
        std::ofstream file(temp_path, std::ios::binary);

        // Header
        uint32_t magic = GGUF_MAGIC;
        file.write(reinterpret_cast<const char*>(&magic), 4);
        uint32_t version = 3;
        file.write(reinterpret_cast<const char*>(&version), 4);
        uint64_t tensor_count = 0;
        file.write(reinterpret_cast<const char*>(&tensor_count), 8);
        uint64_t metadata_count = 5;
        file.write(reinterpret_cast<const char*>(&metadata_count), 8);

        // Metadata 1: UINT32
        write_gguf_string(file, "test.uint32");
        uint32_t type_uint32 = static_cast<uint32_t>(GGUFType::UINT32);
        file.write(reinterpret_cast<const char*>(&type_uint32), 4);
        uint32_t val_uint32 = 42;
        file.write(reinterpret_cast<const char*>(&val_uint32), 4);

        // Metadata 2: FLOAT32
        write_gguf_string(file, "test.float32");
        uint32_t type_float = static_cast<uint32_t>(GGUFType::FLOAT32);
        file.write(reinterpret_cast<const char*>(&type_float), 4);
        float val_float = 3.14159f;
        file.write(reinterpret_cast<const char*>(&val_float), 4);

        // Metadata 3: BOOL
        write_gguf_string(file, "test.bool");
        uint32_t type_bool = static_cast<uint32_t>(GGUFType::BOOL);
        file.write(reinterpret_cast<const char*>(&type_bool), 4);
        uint8_t val_bool = 1;
        file.write(reinterpret_cast<const char*>(&val_bool), 1);

        // Metadata 4: STRING
        write_gguf_string(file, "test.string");
        uint32_t type_string = static_cast<uint32_t>(GGUFType::STRING);
        file.write(reinterpret_cast<const char*>(&type_string), 4);
        write_gguf_string(file, "hello world");

        // Metadata 5: UINT64
        write_gguf_string(file, "test.uint64");
        uint32_t type_uint64 = static_cast<uint32_t>(GGUFType::UINT64);
        file.write(reinterpret_cast<const char*>(&type_uint64), 4);
        uint64_t val_uint64 = 0xDEADBEEFCAFEBABE;
        file.write(reinterpret_cast<const char*>(&val_uint64), 8);
    }

    auto result = GGUFFile::open(temp_path.string());
    REQUIRE(result.ok());

    auto& gguf = result.value();

    // Check metadata keys
    auto keys = gguf.metadata_keys();
    REQUIRE(keys.size() == 5);

    // Check individual values
    REQUIRE(gguf.has_metadata("test.uint32"));
    REQUIRE(gguf.get_metadata_as<uint32_t>("test.uint32") == 42);

    REQUIRE(gguf.has_metadata("test.float32"));
    auto float_val = gguf.get_metadata_as<float>("test.float32");
    REQUIRE(float_val.has_value());
    REQUIRE_THAT(*float_val, Catch::Matchers::WithinRel(3.14159f, 0.0001f));

    REQUIRE(gguf.has_metadata("test.bool"));
    REQUIRE(gguf.get_metadata_as<bool>("test.bool") == true);

    REQUIRE(gguf.has_metadata("test.string"));
    REQUIRE(gguf.get_metadata_as<std::string>("test.string") == "hello world");

    REQUIRE(gguf.has_metadata("test.uint64"));
    REQUIRE(gguf.get_metadata_as<uint64_t>("test.uint64") == 0xDEADBEEFCAFEBABE);

    // Non-existent key
    REQUIRE_FALSE(gguf.has_metadata("nonexistent"));
    REQUIRE_FALSE(gguf.get_metadata("nonexistent").has_value());

    // Type mismatch returns nullopt
    REQUIRE_FALSE(gguf.get_metadata_as<int32_t>("test.string").has_value());

    std::filesystem::remove(temp_path);
}

TEST_CASE("GGUFFile move semantics", "[gguf]") {
    auto temp_path = std::filesystem::temp_directory_path() / "test_move.gguf";

    {
        std::ofstream file(temp_path, std::ios::binary);
        uint32_t magic = GGUF_MAGIC;
        file.write(reinterpret_cast<const char*>(&magic), 4);
        uint32_t version = 3;
        file.write(reinterpret_cast<const char*>(&version), 4);
        uint64_t tensor_count = 0;
        file.write(reinterpret_cast<const char*>(&tensor_count), 8);
        uint64_t metadata_count = 0;
        file.write(reinterpret_cast<const char*>(&metadata_count), 8);
    }

    auto result = GGUFFile::open(temp_path.string());
    REQUIRE(result.ok());

    // Move construction
    GGUFFile file1 = std::move(result).take();
    REQUIRE(file1.is_open());
    REQUIRE(file1.version() == 3);

    // Move assignment
    GGUFFile file2;
    REQUIRE_FALSE(file2.is_open());
    file2 = std::move(file1);
    REQUIRE(file2.is_open());
    REQUIRE_FALSE(file1.is_open());  // NOLINT(bugprone-use-after-move)

    std::filesystem::remove(temp_path);
}

TEST_CASE("GGUFFile summary", "[gguf]") {
    auto temp_path = std::filesystem::temp_directory_path() / "test_summary.gguf";

    {
        std::ofstream file(temp_path, std::ios::binary);

        // Header
        uint32_t magic = GGUF_MAGIC;
        file.write(reinterpret_cast<const char*>(&magic), 4);
        uint32_t version = 3;
        file.write(reinterpret_cast<const char*>(&version), 4);
        uint64_t tensor_count = 1;
        file.write(reinterpret_cast<const char*>(&tensor_count), 8);
        uint64_t metadata_count = 1;
        file.write(reinterpret_cast<const char*>(&metadata_count), 8);

        // Metadata
        write_gguf_string(file, "general.architecture");
        uint32_t value_type = static_cast<uint32_t>(GGUFType::STRING);
        file.write(reinterpret_cast<const char*>(&value_type), 4);
        write_gguf_string(file, "llama");

        // Tensor
        write_gguf_string(file, "weight");
        uint32_t n_dims = 2;
        file.write(reinterpret_cast<const char*>(&n_dims), 4);
        uint64_t dim0 = 4096;
        file.write(reinterpret_cast<const char*>(&dim0), 8);
        uint64_t dim1 = 4096;
        file.write(reinterpret_cast<const char*>(&dim1), 8);
        uint32_t type = static_cast<uint32_t>(GGMLType::Q4_0);
        file.write(reinterpret_cast<const char*>(&type), 4);
        uint64_t offset = 0;
        file.write(reinterpret_cast<const char*>(&offset), 8);
    }

    auto result = GGUFFile::open(temp_path.string());
    REQUIRE(result.ok());

    auto summary = result.value().summary();
    REQUIRE(summary.find("GGUF File") != std::string::npos);
    REQUIRE(summary.find("Version: 3") != std::string::npos);
    REQUIRE(summary.find("Architecture: llama") != std::string::npos);
    REQUIRE(summary.find("Tensors: 1") != std::string::npos);
    REQUIRE(summary.find("Q4_0") != std::string::npos);

    std::filesystem::remove(temp_path);
}

// =============================================================================
// GGUF Constants Tests
// =============================================================================

TEST_CASE("GGUF constants", "[gguf]") {
    REQUIRE(GGUF_MAGIC == 0x46554747);  // "GGUF" in little-endian
    REQUIRE(GGUF_VERSION_MIN == 2);
    REQUIRE(GGUF_VERSION_MAX == 3);
}

#include <catch2/catch_test_macros.hpp>
#include <granite/granite.h>

using namespace granite;

TEST_CASE("DataType size calculations", "[types]") {
    SECTION("FP32 is 4 bytes") {
        REQUIRE(dtype_size(DataType::FP32) == 4);
    }

    SECTION("FP16 is 2 bytes") {
        REQUIRE(dtype_size(DataType::FP16) == 2);
    }

    SECTION("INT8 is 1 byte") {
        REQUIRE(dtype_size(DataType::INT8) == 1);
    }

    SECTION("INT4 is 1 byte (packed)") {
        REQUIRE(dtype_size(DataType::INT4) == 1);
    }
}

TEST_CASE("DataType classification", "[types]") {
    SECTION("Float types") {
        REQUIRE(dtype_is_float(DataType::FP32) == true);
        REQUIRE(dtype_is_float(DataType::FP16) == true);
        REQUIRE(dtype_is_float(DataType::BF16) == true);
        REQUIRE(dtype_is_float(DataType::INT8) == false);
    }

    SECTION("Quantized types") {
        REQUIRE(dtype_is_quantized(DataType::INT4) == true);
        REQUIRE(dtype_is_quantized(DataType::INT8) == true);
        REQUIRE(dtype_is_quantized(DataType::FP32) == false);
    }
}

TEST_CASE("Handle validity", "[types]") {
    SECTION("Default handles are invalid") {
        BufferHandle buf;
        PipelineHandle pipe;
        FenceHandle fence;

        REQUIRE(buf.valid() == false);
        REQUIRE(pipe.valid() == false);
        REQUIRE(fence.valid() == false);
    }

    SECTION("Non-zero handles are valid") {
        BufferHandle buf{1};
        REQUIRE(buf.valid() == true);
    }
}

TEST_CASE("BufferDesc factory methods", "[types]") {
    SECTION("device buffer") {
        auto desc = BufferDesc::device(1024);
        REQUIRE(desc.size == 1024);
        REQUIRE(desc.memory_type == MemoryType::Device);
    }

    SECTION("shared buffer") {
        auto desc = BufferDesc::shared(2048);
        REQUIRE(desc.size == 2048);
        REQUIRE(desc.memory_type == MemoryType::Shared);
    }
}
